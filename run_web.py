#!/usr/bin/env python3
"""
ウェブダッシュボード起動スクリプト

Usage:
  python run_web.py              # http://localhost:8000 で起動
  python run_web.py --port 9000  # ポート変更
  python run_web.py --demo       # モックデータ強制 (ネット不要)
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    p = argparse.ArgumentParser(description="マルチアセット売買指標ダッシュボード")
    p.add_argument("--port",  type=int, default=8000)
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--demo",  action="store_true", help="モックデータ強制")
    p.add_argument("--reload", action="store_true", help="開発用ホットリロード")
    args = p.parse_args()

    if args.demo:
        import data.fetcher as _f
        _f.FORCE_MOCK = True
        print("★ デモモード: モックデータを使用")

    print(f"🚀 ダッシュボード起動中: http://{args.host}:{args.port}")
    print("   Ctrl+C で停止")

    import uvicorn
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
