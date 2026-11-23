# Prometheus service metrics

Prometheus scrapes the API metrics endpoint every 30 seconds. The service publishes request count, request duration histogram, and database connection pool saturation. A high pool saturation ratio combined with rising request duration can indicate a PostgreSQL bottleneck. Alert rules require a sustained condition rather than a single short spike.

Metric labels must have bounded cardinality. The application uses route templates rather than raw URL paths as labels so order IDs cannot create an unbounded number of time series.
