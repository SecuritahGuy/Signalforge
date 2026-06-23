from __future__ import annotations

from signalforge.mlx_backend import mlx_status


def main() -> None:
    status = mlx_status()
    if status["available"]:
        print("MLX available: Apple silicon Metal backend is accessible.")
    else:
        print("MLX unavailable in this process.")
        print(status["error"])


if __name__ == "__main__":
    main()
