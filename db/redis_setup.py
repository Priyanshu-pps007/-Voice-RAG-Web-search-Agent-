import redis

# 1. Connect to Redis (decode_responses=True automatically converts bytes to strings)
rag_redis = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

