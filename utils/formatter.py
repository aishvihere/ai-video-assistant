def format_items(items, title: str = "") -> str:
    lines = []
    if title:
        lines += ["=" * 60, title, "=" * 60]
    if not items:
        lines.append("(none found)")
        return "\n".join(lines)

    for n, item in enumerate(items, 1):
        data = {k: v for k, v in item.model_dump().items() if v not in ("", None, [])}
        fields = list(data.items())
        _, first_val = fields[0]              # first field becomes the headline
        lines.append(f"{n}. {first_val}")
        for key, val in fields[1:]:
            label = key.replace("_", " ").capitalize()
            lines.append(f"   {label}: {val}")
        lines.append("")
    return "\n".join(lines)