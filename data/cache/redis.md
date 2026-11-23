# Redis cache invalidation

Redis stores short-lived derived responses. The order-summary cache has a time-to-live of 60 seconds. A write commits the canonical record in PostgreSQL before it invalidates the corresponding Redis key. If the transaction rolls back, the key is not invalidated. When Redis is unavailable, the API falls back to reading PostgreSQL and logs a cache warning rather than losing order data.

Cache keys include the order ID and a schema version so a deployment does not deserialize an incompatible old value. Operators monitor cache hit rate and database load during an outage.
