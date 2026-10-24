"""Build quantdesk_app.py: the app template with the quantdesk package packed inside.

    python tools/build_app.py           # rebuild
    python tools/build_app.py --check   # fail if quantdesk_app.py is out of date (used by CI)
"""
import base64
import io
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        pkg = os.path.join(ROOT, "quantdesk")
        for f in sorted(os.listdir(pkg)):
            if f.endswith(".py"):
                info = zipfile.ZipInfo("quantdesk/" + f, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(os.path.join(pkg, f), "rb") as fh:
                    z.writestr(info, fh.read())
    bundle = base64.b64encode(buf.getvalue()).decode()
    with open(os.path.join(ROOT, "tools", "app_template.py"), encoding="utf-8") as fh:
        template = fh.read()
    return template.replace("__BUNDLE__", bundle, 1)


if __name__ == "__main__":
    out = os.path.join(ROOT, "quantdesk_app.py")
    text = build()
    if "--check" in sys.argv:
        with open(out, encoding="utf-8") as fh:
            ok = fh.read() == text
        print("quantdesk_app.py is up to date" if ok else "quantdesk_app.py is stale: run tools/build_app.py")
        sys.exit(0 if ok else 1)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {out} ({len(text) / 1024:.0f} KB)")
