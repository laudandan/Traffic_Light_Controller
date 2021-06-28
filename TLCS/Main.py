from training_main import train_main
from testing_main import test_main
import sys
# 1x1 - 2 agenti inteligenti
# 1x1dumb - 1 agent inteligent 1 dumb
# 2x2 - 4 agenti inteligenti

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    if len(sys.argv) > 2:
        print("Too much arguments introduced")
    else:
        for i, arg in enumerate(sys.argv):
            print(f"Argument {i:>2}: {arg}")
    # train_main(model="1x1dumb")
    test_main(model="1x1dumb")
    print("Done")