#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
import feedforward

if __name__ == '__main__':
    main(MyDriver())
