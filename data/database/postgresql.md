# PostgreSQL connection pool

The application limits its PostgreSQL pool to 20 concurrent connections per API instance. With three instances, the theoretical application maximum is 60 database connections before administrative or background connections. When the database reports too many connections, operators compare active connections with the pool size and instance count before increasing either limit.

Transactions protect order writes. The API commits an order before publishing an invalidation event. A failed transaction rolls back and must not cause a cache invalidation or a success response.
