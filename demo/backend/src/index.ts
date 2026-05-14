import { createApp } from "./server.js";

const port = Number(process.env.PORT ?? 8080);
const host = process.env.HOST ?? "0.0.0.0";

createApp().listen(port, host, () => {
  console.log(`oil-signal-mine-backend listening on http://${host}:${port}`);
});
