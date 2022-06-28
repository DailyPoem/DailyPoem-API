import uvicorn
import sys, getopt

from configparser import ConfigParser


def main(argv: list() = sys.argv):
    """
    reload 옵션과 PORT를 지정해주는 함수
    """

    config = ConfigParser()
    config.read("config.ini")

    FILE_NAME = argv[0]
    reload: str = "False"
    try:
        PORT: int = int(config.get("default", "PORT"))
    except Exception as e:
        print(e)
        PORT = 5000

    try:
        opts, _ = getopt.getopt(argv[1:], "HRP:", ["reload", "PORT="])

    except getopt.GetoptError:
        print(FILE_NAME, "-R -P <Port Number>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-H", "--help"):
            print(FILE_NAME, "-R -P <Port Number>")
            sys.exit(2)
        elif opt in ("-R", "--reload"):
            reload: str = "True"
            # "reload setting option : -R, --reload"
        elif opt in ("-P", "--PORT"):
            PORT: int = int(arg)
            # "PORT setting option : -P args, --PORT=args"
    uvicorn.run("src:app", host="0.0.0.0", port=PORT, reload=reload)


if __name__ == "__main__":
    main(sys.argv)