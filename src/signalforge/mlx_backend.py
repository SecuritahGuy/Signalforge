from __future__ import annotations


def mlx_status() -> dict:
    """Return MLX availability without making the core pipeline depend on it."""
    try:
        import mlx.core as mx

        test_value = mx.array([1.0, 2.0]).sum()
        mx.eval(test_value)
    except Exception as exc:  # noqa: BLE001 - status probe should report any backend failure.
        return {
            "available": False,
            "error": str(exc),
        }
    return {
        "available": True,
        "device": "apple_silicon_metal",
    }
