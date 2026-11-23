# Nginx reverse proxy

Nginx terminates TLS and forwards requests to healthy API instances. It preserves the request ID header and applies a request-body size limit before forwarding uploads. The proxy returns a gateway error if no upstream instance is healthy.
