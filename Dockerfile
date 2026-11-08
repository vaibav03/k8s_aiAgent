# KubeAI agent image (Node 20 + pinned kubectl for allowlisted remediation).
# Build straight into the minikube node:  minikube -p kubeai-lab image build -t kubeai-agent:latest .
FROM node:20-alpine

ARG KUBECTL_VERSION=v1.33.1
# kubectl for the build platform's arch (works with BuildKit and with the classic builder that
# `minikube image build` may use, where TARGETARCH is not set)
RUN apk add --no-cache curl ca-certificates \
 && case "$(apk --print-arch)" in x86_64) ARCH=amd64 ;; aarch64) ARCH=arm64 ;; *) echo "unsupported arch" >&2; exit 1 ;; esac \
 && curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${ARCH}/kubectl" \
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
