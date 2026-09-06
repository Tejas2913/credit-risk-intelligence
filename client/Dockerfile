# ==============================================================================
# Credit Risk Intelligence Platform - Frontend Multi-Stage Dockerfile
# Stage 1: Build production static bundle using Node.js
# Stage 2: Serve optimized assets with Nginx reverse proxy
# ==============================================================================

# --- Stage 1: Build ---
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency specifications and install cleanly with npm ci
COPY package.json package-lock.json ./
RUN npm ci

# Copy full frontend source code
COPY . .

# Build production bundle with Vite
RUN npm run build

# --- Stage 2: Production Nginx Server ---
FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy built frontend assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx reverse proxy configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose HTTP port
EXPOSE 80

# Health check
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
