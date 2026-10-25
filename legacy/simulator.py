import math


def clamp(value, minimum, maximum):
    """将数值限制在指定范围内。"""
    return max(minimum, min(value, maximum))


def simulate_treatment(scene, aeration, recycle_ratio):
    """
    模拟 24 小时污水处理过程。

    scene: "normal" 或 "shock"
    aeration: 曝气量，建议范围 1.0 到 6.0
    recycle_ratio: 回流比，建议范围 0.5 到 2.0
    """
    aeration = clamp(float(aeration), 1.0, 6.0)
    recycle_ratio = clamp(float(recycle_ratio), 0.5, 2.0)

    is_shock = scene == "shock"
    data = []

    for hour in range(24):
        # 用正弦变化模拟一天中进水水质的自然波动。
        wave = math.sin(hour / 24 * math.pi * 2)

        influent_cod = 280 + wave * 25
        influent_ammonia = 30 + wave * 4

        # 冲击负荷场景下，8 至 15 点进水污染物明显升高。
        if is_shock and 8 <= hour <= 15:
            influent_cod += 180
            influent_ammonia += 22

        # 简化规则：曝气量和回流比提高，会增强 COD、氨氮去除效果。
        cod_removal_rate = clamp(
            0.52 + aeration * 0.055 + recycle_ratio * 0.035,
            0.45,
            0.88,
        )
        ammonia_removal_rate = clamp(
            0.20 + aeration * 0.105 + recycle_ratio * 0.030,
            0.15,
            0.90,
        )

        cod = influent_cod * (1 - cod_removal_rate)
        ammonia = influent_ammonia * (1 - ammonia_removal_rate)

        # 曝气越大，DO 越高；冲击负荷会额外消耗氧气。
        oxygen_consumption = 0.6 if is_shock and 8 <= hour <= 15 else 0.0
        dissolved_oxygen = clamp(
            0.45 + aeration * 0.62 - oxygen_consumption + wave * 0.12,
            0.2,
            6.5,
        )

        # MLSS 仅作教学用途的简化模拟值。
        mlss = 2850 + recycle_ratio * 260 - aeration * 35 + wave * 80

        data.append(
            {
                "hour": hour,
                "influent_cod": round(influent_cod, 1),
                "influent_ammonia": round(influent_ammonia, 1),
                "cod": round(cod, 1),
                "ammonia": round(ammonia, 1),
                "do": round(dissolved_oxygen, 2),
                "mlss": round(mlss, 1),
            }
        )

    current = data[-1]
    alerts = []
    advice = "当前工艺参数基本稳定。"

    if current["cod"] > 70:
        alerts.append("出水 COD 偏高")
        advice = "建议适当提高曝气量，或检查进水负荷。"

    if current["ammonia"] > 8:
        alerts.append("出水氨氮偏高")
        advice = "建议提高曝气量，增强硝化效果。"

    if current["do"] < 1.5:
        alerts.append("溶解氧 DO 偏低")
        advice = "建议提高曝气量，避免缺氧。"

    return {
        "scene": "冲击负荷" if is_shock else "正常进水",
        "parameters": {
            "aeration": aeration,
            "recycle_ratio": recycle_ratio,
        },
        "current": current,
        "alerts": alerts,
        "advice": advice,
        "series": data,
    }