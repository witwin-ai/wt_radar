"""Plugin-owned Agent retrieval vocabulary for independent Radar steps."""

RADAR_PACK_TAGS = (
    "pack:witwin.radar-pipeline",
    "priority:10",
    "word-intent:radar simulation",
    "word-intent:radar measurement",
    "word-intent:range profile",
    "word-intent:range doppler",
    "word-intent:radar result",
    "word-intent:radar export",
    "word-intent:saved recording",
    "word-intent:previous recording",
    "word-intent:existing result",
    "word-intent:same result",
    "word-intent:last successful recording",
    "word-intent:recording we already have",
    "intent:雷达仿真",
    "intent:雷达测量",
    "intent:距离像",
    "intent:距离多普勒",
    "intent:导出雷达结果",
)


def hints(*terms: str, boost: int = 18) -> tuple[str, ...]:
    return RADAR_PACK_TAGS + tuple(f"hint-term:{term}" for term in terms) + (
        f"hint-boost:{boost}",
    )


def actions(*intents: str, boost: int = 19) -> tuple[str, ...]:
    return RADAR_PACK_TAGS + tuple(
        f"hint-explicit-intent:{intent}" for intent in intents
    ) + (
        "hint-requires-pack",
        f"hint-boost:{boost}",
        "hint-explicit-only",
    )


INSPECT_TAGS = hints("radar", "radar result", "雷达", "雷达结果", boost=16)
# Result discovery is read-only and does not make runtime diagnostics or writers
# relevant merely because a user asks about a saved recording or an export.
RESULT_INSPECT_TAGS = INSPECT_TAGS + tuple(f"hint-term:{term}" for term in (
    "saved recording", "recorded result", "existing result", "replay", "play back",
    "npz", "export", "download", "已有结果", "保存的录制", "回放", "导出",
))
PLAN_TAGS = hints(
    "plan radar", "radar preflight", "check radar simulation",
    "rerun radar", "measure again", "current radar position", "moved radar",
    "use where i moved the radar", "keep the room cat and motion unchanged",
    "雷达预检", "重新测量", "当前位置", "移动雷达", boost=18,
)
SENSOR_PLAN_TAGS = hints(
    "plan radar position", "plan radar placement", "radar position", "radar placement",
    "point radar", "closer radar", "farther radar", "whole path visible",
    "approaching and receding", "stronger doppler", "side view", "opposite side",
    "雷达位置", "规划雷达位置", "雷达摆位", "靠近", "远离", "完整路径可见",
    "接近和远离", "更强多普勒", "侧面", "另一侧", boost=20,
)
ENSURE_TAGS = actions(
    "place radar", "configure radar", "move radar", "closer radar", "farther radar",
    "whole path visible", "approaching and receding", "stronger doppler", "side view",
    "opposite side", "放置雷达", "配置雷达", "移动雷达", "靠近", "远离",
    "完整路径可见", "接近和远离", "更强多普勒", "侧面", "另一侧", boost=19,
)
SUBMIT_TAGS = actions(
    "run only radar", "run radar simulation", "start radar simulation",
    "run the simulation again", "use where i moved the radar",
    "只运行 Radar", "只运行雷达", "开始雷达仿真", boost=20,
)
STATUS_TAGS = hints("radar status", "simulation status", "雷达状态", "仿真状态", boost=17)
CANCEL_TAGS = actions("cancel radar simulation", "取消雷达仿真", boost=20)
VERIFY_TAGS = hints("verify radar", "verify result", "验证雷达", "验证结果", boost=18)
REPLAY_TAGS = actions(
    "prepare radar replay", "replay only", "show radar on timeline",
    "replay", "watch", "play back",
    "只回放", "准备雷达回放", "同步雷达预览", boost=18,
)
EXPORT_TAGS = actions(
    "export radar", "export only", "save radar npz",
    "export", "download", "npz",
    "只导出", "导出雷达", "保存雷达 NPZ", boost=19,
)
