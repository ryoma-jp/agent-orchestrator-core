FROM node:22-alpine AS build
WORKDIR /src
ARG APP_DIR=user
COPY ui/${APP_DIR}/package.json ui/${APP_DIR}/package-lock.json ./
RUN npm ci
COPY ui/${APP_DIR}/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY ui/nginx-app.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
