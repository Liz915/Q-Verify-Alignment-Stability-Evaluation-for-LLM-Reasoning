import os
import runpy


def main():
    print("[INFO] `experiments/run_alignment.py` is a compatibility wrapper.")
    print("[INFO] Forwarding to `experiments/train_real_dpo.py`.")
    target = os.path.join(os.path.dirname(__file__), "train_real_dpo.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
