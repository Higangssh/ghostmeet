from .app import app
from .config import config
import uvicorn


def main() -> None:
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
