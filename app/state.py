# In-memory storage for chat state
# Warning: This is not suitable for production environments with multiple workers/instances.
# Consider using a database or distributed cache for persistence and scalability.

chat_histories = {} # {session_id: [messages]}
user_details = {}   # {session_id: { ... state details ... }} 