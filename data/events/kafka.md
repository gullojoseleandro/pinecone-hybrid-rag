# Kafka event ordering

Kafka events use the order ID as the partition key. Events for one order therefore reach the same partition and preserve their relative order. Consumers keep an offset checkpoint after processing succeeds. A consumer may see a message again after a crash, so its handler is idempotent.
