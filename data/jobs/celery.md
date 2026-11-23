# Celery task retries

Celery workers process notification tasks outside the HTTP request path. A notification task uses exponential retry delay for transient network errors and has a maximum of three attempts. The task includes an idempotency key derived from the order ID and notification type so a retry does not send duplicate messages.

Permanent validation errors are not retried. Operators inspect the dead-letter queue when all attempts fail and correlate task logs using the request ID.
