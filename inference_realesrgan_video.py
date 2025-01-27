# inference_realesrgan_video.py
import argparse
import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg

def fix_path(path):
    """Fix path for Windows compatibility"""
    return os.path.normpath(path).replace('\\', '/')

def ensure_path_exists(path, create=False):
    """Check if path exists and optionally create it"""
    path = fix_path(path)
    if create and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    elif not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    return path

def verify_cuda():
    """Verify CUDA is available and working"""
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA is not available. Installing PyTorch with CUDA...")
        print("Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False

    try:
        # Try to allocate a small tensor on GPU
        x = torch.rand(1).cuda()
        del x  # Free memory
        torch.cuda.empty_cache()  # Clear GPU cache
        return True
    except Exception as e:
        print(f"\nWARNING: CUDA error: {e}")
        return False

def setup_gpu():
    """Setup and verify GPU configuration"""
    cuda_available = verify_cuda()
    if cuda_available:
        # Set default GPU if not specified
        if torch.cuda.current_device() != 0:
            torch.cuda.set_device(0)
        # Enable TF32 if available for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    return False

def print_device_info():
    """Print detailed device information"""
    print("\nDevice Information:")
    cuda_available = verify_cuda()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU Index: {torch.cuda.current_device()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU detected - running on CPU")
        print("Please install PyTorch with CUDA support for GPU acceleration")

def get_video_meta_info(video_path):
    """Get video metadata with improved error handling"""
    try:
        video_path = ensure_path_exists(video_path)
        ret = {}
        probe = ffmpeg.probe(video_path)
        video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        if not video_streams:
            raise ValueError("No video stream found")

        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        ret['width'] = video_streams[0]['width']
        ret['height'] = video_streams[0]['height']
        ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
        ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
        ret['nb_frames'] = int(video_streams[0]['nb_frames'])
        return ret
    except Exception as e:
        print(f"Error reading video metadata: {e}")
        raise

def get_sub_video(args, num_process, process_idx):
    """Split video into sub-videos for parallel processing"""
    if num_process == 1:
        return ensure_path_exists(args.input)

    try:
        meta = get_video_meta_info(args.input)
        duration = int(meta['nb_frames'] / meta['fps'])
        part_time = duration // num_process
        print(f'Video duration: {duration}s, Part duration: {part_time}s')

        out_dir = ensure_path_exists(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), create=True)
        out_path = fix_path(osp.join(out_dir, f'{process_idx:03d}.mp4'))

        cmd = [
            args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
            f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '',
            '-async 1', out_path, '-y'
        ]
        print(f'Splitting video: {" ".join(cmd)}')
        subprocess.call(' '.join(cmd), shell=True)
        return out_path
    except Exception as e:
        print(f"Error splitting video: {e}")
        return ensure_path_exists(args.input)

class Reader:
    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        self.input_path = ensure_path_exists(args.input)
        input_type = mimetypes.guess_type(self.input_path)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []
        self.audio = None
        self.input_fps = None

        if self.input_type.startswith('video'):
            self._init_video(total_workers, worker_idx)
        else:
            self._init_images(total_workers, worker_idx)

    def _init_video(self, total_workers, worker_idx):
        """Initialize video reader"""
        video_path = get_sub_video(self.args, total_workers, worker_idx)
        self.stream_reader = (
            ffmpeg.input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
            .run_async(pipe_stdin=True, pipe_stdout=True, cmd=self.args.ffmpeg_bin)
        )
        meta = get_video_meta_info(video_path)
        self.width = meta['width']
        self.height = meta['height']
        self.input_fps = meta['fps']
        self.audio = meta['audio']
        self.nb_frames = meta['nb_frames']

    def _init_images(self, total_workers, worker_idx):
        """Initialize image reader"""
        if self.input_type.startswith('image'):
            self.paths = [self.input_path]
        else:
            all_paths = sorted(glob.glob(os.path.join(self.input_path, '*')))
            tot_frames = len(all_paths)
            num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
            self.paths = all_paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

        self.nb_frames = len(self.paths)
        if self.nb_frames == 0:
            raise ValueError('No images found in input folder')

        # Get image dimensions from first image
        from PIL import Image
        with Image.open(self.paths[0]) as img:
            self.width, self.height = img.size

        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        """Read frame from video stream"""
        try:
            img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)
            if not img_bytes:
                return None
            return np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        except Exception as e:
            print(f"Error reading video frame: {e}")
            return None

    def get_frame_from_list(self):
        """Read frame from image list"""
        if self.idx >= self.nb_frames:
            return None
        try:
            img = cv2.imread(self.paths[self.idx])
            self.idx += 1
            return img
        except Exception as e:
            print(f"Error reading image frame: {e}")
            self.idx += 1
            return None

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video') and hasattr(self, 'stream_reader'):
            try:
                self.stream_reader.stdin.close()
                self.stream_reader.wait()
            except Exception as e:
                print(f"Error closing video stream: {e}")

class Writer:
    def __init__(self, args, audio, height, width, video_save_path, fps):
        """Initialize video writer with error handling"""
        try:
            self.out_width = int(width * args.outscale)
            self.out_height = int(height * args.outscale)

            if self.out_height > 2160:
                print('Warning: Output resolution exceeds 4K.',
                      'This will be very slow. Consider reducing outscale.')

            # Ensure output directory exists
            os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

            if audio is not None:
                self._init_with_audio(args, audio, video_save_path, fps)
            else:
                self._init_without_audio(args, video_save_path, fps)

        except Exception as e:
            print(f"Error initializing video writer: {e}")
            raise

    def _init_with_audio(self, args, audio, video_save_path, fps):
        """Initialize writer with audio"""
        self.stream_writer = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24',
                        s=f'{self.out_width}x{self.out_height}', framerate=fps)
            .output(audio, video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                   loglevel='error', acodec='copy')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
        )

    def _init_without_audio(self, args, video_save_path, fps):
        """Initialize writer without audio"""
        self.stream_writer = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24',
                        s=f'{self.out_width}x{self.out_height}', framerate=fps)
            .output(video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                   loglevel='error')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
        )

    def write_frame(self, frame):
        """Write frame with error handling"""
        try:
            frame = frame.astype(np.uint8).tobytes()
            self.stream_writer.stdin.write(frame)
        except Exception as e:
            print(f"Error writing frame: {e}")

    def close(self):
        """Close writer with error handling"""
        try:
            self.stream_writer.stdin.close()
            self.stream_writer.wait()
        except Exception as e:
            print(f"Error closing writer: {e}")

def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    """Run video inference with improved error handling and GPU management"""
    try:
        # Setup device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if str(device) == 'cuda':
            torch.cuda.set_device(device)
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")

        args.model_name = args.model_name.split('.pth')[0]

        # Model architecture selection
        model, netscale, file_url = setup_model(args)

        # Model paths
        model_path = setup_model_path(args, file_url)

        # Initialize upsampler
        upsampler = initialize_upsampler(args, model, model_path, device)

        # Setup face enhancer if requested
        face_enhancer = setup_face_enhancer(args, upsampler) if args.face_enhance else None

        # Initialize reader and writer
        reader = Reader(args, total_workers, worker_idx)
        audio = reader.get_audio()
        height, width = reader.get_resolution()
        fps = reader.get_fps()
        writer = Writer(args, audio, height, width, video_save_path, fps)

        # Process frames
        process_frames(args, reader, writer, upsampler, face_enhancer, device)

        # Cleanup
        reader.close()
        writer.close()

    except Exception as e:
        print(f"Error during inference: {e}")
        raise

def setup_model(args):
    """Setup model architecture based on model name"""
    if args.model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f'Unknown model name: {args.model_name}')

    return model, netscale, file_url

def setup_model_path(args, file_url):
    """Setup and download model path if necessary"""
    weights_dir = ensure_path_exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights'), create=True)
    model_path = os.path.join(weights_dir, args.model_name + '.pth')

    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            model_path = load_file_from_url(
                url=url,
                model_dir=os.path.join(ROOT_DIR, 'weights'),
                progress=True,
                file_name=None
            )

    # Setup DNI weights if needed
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        return [model_path, wdn_model_path], [args.denoise_strength, 1 - args.denoise_strength]

    return model_path, None

def initialize_upsampler(args, model, model_path, device):
    """Initialize upsampler with model"""
    dni_weight = None
    if isinstance(model_path, list):
        dni_weight = model_path[1]
        model_path = model_path[0]

    upsampler = RealESRGANer(
        scale=args.outscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=device,
    )

    if 'anime' in args.model_name and args.face_enhance:
        print('Face enhancement is not supported in anime models - option disabled.')
        args.face_enhance = False

    return upsampler

def setup_face_enhancer(args, upsampler):
    """Setup GFPGAN face enhancer if requested"""
    try:
        from gfpgan import GFPGANer
        return GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )
    except Exception as e:
        print(f"Error setting up face enhancer: {e}")
        return None

def process_frames(args, reader, writer, upsampler, face_enhancer, device):
    """Process video frames with progress bar"""
    with tqdm(total=len(reader), unit='frame', desc='Processing') as pbar:
        while True:
            img = reader.get_frame()
            if img is None:
                break

            try:
                if args.face_enhance and face_enhancer is not None:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = upsampler.enhance(img, outscale=args.outscale)

                writer.write_frame(output)
                if device is not None and str(device) != 'cpu':
                    torch.cuda.synchronize(device)

            except RuntimeError as error:
                print(f'\nError processing frame: {error}')
                print('If encountering CUDA out of memory, try setting --tile to a smaller number.')
                continue

            pbar.update(1)

            # Optional: Show current GPU memory usage
            if device is not None and str(device) != 'cpu':
                pbar.set_postfix({'GPU Memory': f'{torch.cuda.memory_allocated(device)/1024**2:.1f}MB'})

def run(args):
    """Main execution function with improved error handling"""
    try:
        # Setup and verify GPU
        setup_gpu()
        print_device_info()

        # Fix and verify paths
        args.input = ensure_path_exists(args.input)
        args.output = ensure_path_exists(args.output, create=True)
        args.video_name = osp.splitext(os.path.basename(args.input))[0]
        video_save_path = fix_path(osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4'))

        # Handle frame extraction if requested
        if args.extract_frame_first:
            tmp_frames_folder = ensure_path_exists(
                osp.join(args.output, f'{args.video_name}_inp_tmp_frames'),
                create=True
            )
            os.system(f'ffmpeg -i {args.input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 {tmp_frames_folder}/frame%08d.png')
            args.input = tmp_frames_folder

        # Setup processing mode (single/multi GPU)
        num_gpus = torch.cuda.device_count()
        print(f"\nUsing {num_gpus} GPU(s)")

        if num_gpus == 0:
            print("No GPU detected. Running in CPU mode...")
            args.num_process_per_gpu = 1
            inference_video(args, video_save_path)
            return

        num_process = max(1, num_gpus * args.num_process_per_gpu)

        if num_process == 1:
            print("Running in single process mode...")
            inference_video(args, video_save_path)
            return

        # Multi-process mode
        print(f"Running in multi-process mode with {num_process} processes...")
        ctx = torch.multiprocessing.get_context('spawn')
        pool = ctx.Pool(num_process)

        # Create output directories
        os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)

        # Process video segments
        with tqdm(total=num_process, unit='sub_video', desc='Processing') as pbar:
            for i in range(num_process):
                sub_video_save_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
                pool.apply_async(
                    inference_video,
                    args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
                    callback=lambda _: pbar.update(1)
                )

        pool.close()
        pool.join()

        # Combine video segments
        print("\nCombining video segments...")
        vidlist_path = f'{args.output}/{args.video_name}_vidlist.txt'
        with open(vidlist_path, 'w') as f:
            for i in range(num_process):
                f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

        cmd = [
            args.ffmpeg_bin, '-f', 'concat', '-safe', '0',
            '-i', vidlist_path, '-c', 'copy', video_save_path
        ]
        subprocess.call(cmd)

        # Cleanup temporary files
        print("Cleaning up temporary files...")
        cleanup_paths = [
            osp.join(args.output, f'{args.video_name}_out_tmp_videos'),
            osp.join(args.output, f'{args.video_name}_inp_tmp_videos'),
            vidlist_path
        ]
        for path in cleanup_paths:
            if osp.exists(path):
                if osp.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("Falling back to single process mode...")
        try:
            inference_video(args, video_save_path)
        except Exception as e2:
            print(f"Fatal error: {str(e2)}")

def main():
    """Main entry point with argument parsing and error handling"""
    parser = argparse.ArgumentParser(description='Real-ESRGAN Video Inference')

    # Input/Output arguments
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')

    # Model arguments
    parser.add_argument(
        '-n', '--model_name',
        type=str,
        default='realesr-animevideov3',
        help='Model name: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | '
             'RealESRNet_x4plus | RealESRGAN_x2plus | realesr-general-x4v3'
    )

    # Processing arguments
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--num_process_per_gpu', type=int, default=1, help='Number of processes per GPU')
    parser.add_argument('--extract_frame_first', action='store_true', help='Extract frames before processing')

    # Enhancement options
    parser.add_argument(
        '-dn', '--denoise_strength',
        type=float,
        default=0.5,
        help='Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability.'
    )
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic'
    )

    # Other options
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='Path to ffmpeg')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs'
    )

    args = parser.parse_args()

    # Fix paths
    args.input = fix_path(args.input)
    args.output = fix_path(args.output)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Determine input type
    is_video = bool(mimetypes.guess_type(args.input)[0] and
                   mimetypes.guess_type(args.input)[0].startswith('video'))

    # Convert FLV to MP4 if needed
    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        os.system(f'ffmpeg -i {args.input} -codec copy {mp4_path}')
        args.input = mp4_path

    # Disable frame extraction for non-video inputs
    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    try:
        # Run main processing
        run(args)
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nFatal error: {str(e)}')
    finally:
        # Cleanup temporary frame folder if it exists
        if args.extract_frame_first:
            tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
            if osp.exists(tmp_frames_folder):
                try:
                    shutil.rmtree(tmp_frames_folder)
                except Exception as e:
                    print(f"Error cleaning up temporary files: {e}")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()