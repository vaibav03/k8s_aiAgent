# KubeAI agent image (Node 20 + pinned kubectl for allowlisted remediation).
FROM node:20-alpine

ARG KUBECTL_VERSION=v1.31.0
ARG TARGETARCH=amd64
RUN apk add --no-cache curl ca-certificates \
 && curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
 && chmod +x /usr/local/bin/kubectl \
 && apk del curl

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY index.js ./
COPY lib ./lib

ENV NODE_ENV=production PORT=3001
USER node
EXPOSE 3001
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:3001/healthz || exit 1
CMD ["node", "index.js"]
