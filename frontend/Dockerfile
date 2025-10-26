# --------------------------------------------------------------
# 1️⃣  Build stage – compile React (or Vite) app
# --------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci --silent

COPY frontend/ ./
# The build step reads REACT_APP_BACKEND_URL from the build environment.
ARG REACT_APP_BACKEND_URL
ENV REACT_APP_BACKEND_URL=${REACT_APP_BACKEND_URL}
RUN npm run build   # creates ./build

# --------------------------------------------------------------
# 2️⃣  Runtime stage – serve static files with Nginx
# --------------------------------------------------------------
FROM nginx:alpine

# Remove the default server block
RUN rm /etc/nginx/conf.d/default.conf

# Install envsubst (to replace $PORT at container start)
RUN apk add --no-cache gettext

# Custom Nginx config – note the `listen ${PORT};`
COPY <<'EOF' /etc/nginx/conf.d/gaze.conf
server {
    listen ${PORT};
    server_name _;

    # Serve the React SPA
    location / {
        root   /usr/share/nginx/html;
        try_files $uri /index.html =404;
    }
}
EOF

# Render will replace ${PORT} with the actual port at start‑up.
# The CMD runs envsubst once and then starts nginx.
CMD /bin/sh -c "envsubst < /etc/nginx/conf.d/gaze.conf > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"

# Copy the compiled assets from the builder
COPY --from=builder /app/build /usr/share/nginx/html