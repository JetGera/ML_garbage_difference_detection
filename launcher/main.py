try:
    from .gui import main
except ImportError:
    from gui import main


if __name__ == "__main__":
    main()
