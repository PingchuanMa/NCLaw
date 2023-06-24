from pathlib import Path
import subprocess
import shutil


def volume_sampling(
        input_path: Path,
        output_path: Path,
        radius: float,
        minimum: tuple[float, float, float] = (0, 0, 0),
        maximum: tuple[float, float, float] = (1, 1, 1),
        res: tuple[int, int, int] = (30, 30, 30)
    ):

    root = Path(__file__).resolve().parent

    print('{},{},{}'.format(*res))
    subprocess.run([
        str(root / 'extern' / 'VolumeSampling'),
        '-i', str(input_path),
        '-o', str(output_path),
        '-r', str(radius),
        '--region', '{},{},{},{},{},{}'.format(*minimum, *maximum),
        '--res', '{},{},{}'.format(*res)
    ])

    shutil.rmtree(root / 'extern' / 'Cache', ignore_errors=True)
    shutil.rmtree(root / 'extern' / 'output', ignore_errors=True)
