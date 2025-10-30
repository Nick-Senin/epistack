import json
from typing import Dict


def safe_json_dict(raw: str, keys=("A","T","B")) -> Dict[str,str]:
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and all(k in d for k in keys):
            # нормализуем типы
            return {k: str(d.get(k,"")).strip() for k in keys}
    except Exception:
        pass
    # попытка распарсить простые форматы "A:..; T:..; B:.."
    d = {}
    for part in raw.split(";"):
        if ":" in part:
            k,v = part.split(":",1)
            d[k.strip()] = v.strip()
    if all(k in d for k in keys):
        return {k: d[k] for k in keys}
    return {}


def jaccard_like(a: Dict[str,str], b: Dict[str,str]) -> float:
    sa = set(" ".join(a.values()).lower().split())
    sb = set(" ".join(b.values()).lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

