
import logging

logger = logging.getLogger("myLog")

FileOutputHandler = logging.FileHandler('logs.log', mode='w')

#logger.setLevel(level=logging.DEBUG)
logger.setLevel(level=logging.INFO)

formatter = logging.Formatter(fmt='%(levelname)s: %(message)s')

FileOutputHandler.setFormatter(formatter)

logger.addHandler(FileOutputHandler)
logger.propagate = False

