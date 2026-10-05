from app import request_worker
if __name__ == '__main__':
    request_worker({'op': 'ping'}, 2)
