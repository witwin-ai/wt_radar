"""Radar-owned discovery vocabulary for the generic Workbench retriever.

Tags rank available tools; they never grant permissions or replace confirmation.
Keep this module independent of Studio, scene state, and the native solver.
"""

RADAR_PACK_TAGS = (
    "pack:witwin.radar-pipeline",
    "priority:10",
    *(f"word-intent:{term}" for term in (
        "radar simulation", "radar measurement", "radar imaging", "range profile",
        "range doppler", "doppler map", "radar result", "radar export",
        "place radar", "transmitter receiver", "gpu simulation",
    )),
    *(f"intent:{term}" for term in (
        "雷达仿真", "雷达测量", "雷达成像", "距离像", "距离多普勒", "多普勒图",
        "放置雷达", "发射器", "接收器", "导出雷达结果",
    )),
)


def _read_hints(boost: int, *terms: str) -> tuple[str, ...]:
    return (*RADAR_PACK_TAGS, *(f"hint-term:{term}" for term in terms), f"hint-boost:{boost}")


def _action_hints(boost: int, *intents: str) -> tuple[str, ...]:
    return (
        *RADAR_PACK_TAGS,
        *(f"hint-explicit-intent:{intent}" for intent in intents),
        "hint-requires-pack", f"hint-boost:{boost}", "hint-explicit-only",
    )


TOOL_TAGS = {
    "runtime_diagnostics": RADAR_PACK_TAGS,
    "inspect_pipeline": _read_hints(
        16, "radar", "radar simulation", "radar measurement", "range profile", "range doppler",
        "雷达", "雷达仿真", "距离像", "距离多普勒",
    ),
    "plan_animation_measurement": _read_hints(
        18, "plan radar", "radar preflight", "check radar simulation", "gpu preflight",
        "规划雷达", "雷达预检", "检查雷达仿真",
    ),
    "ensure_sensor": _action_hints(
        19, "place radar", "add radar", "configure radar", "prepare radar",
        "put the radar", "position the radar", "放置雷达", "添加雷达", "配置雷达", "把雷达放在",
    ),
    "submit_animation_measurement": _action_hints(
        20, "run radar simulation", "start radar simulation", "simulate animation",
        "run gpu simulation", "take radar measurement", "运行雷达仿真", "开始雷达仿真",
        "仿真动画", "运行 GPU 仿真",
    ),
    "get_simulation": _read_hints(
        17, "radar status", "simulation status", "check radar run", "雷达状态", "仿真状态",
    ),
    "cancel_simulation": _action_hints(
        20, "cancel radar simulation", "stop radar simulation", "取消雷达仿真", "停止雷达仿真",
    ),
    "verify_result": _read_hints(
        18, "verify radar", "verify result", "radar evidence", "验证雷达", "验证结果",
    ),
    "prepare_replay": _action_hints(
        18, "prepare radar replay", "show radar on timeline", "synchronized radar preview",
        "准备雷达回放", "雷达跟随 timeline", "同步雷达预览",
    ),
    "export_result": _action_hints(
        19, "export radar", "download radar result", "save radar npz",
        "导出雷达", "下载雷达结果", "保存雷达 NPZ",
    ),
}
