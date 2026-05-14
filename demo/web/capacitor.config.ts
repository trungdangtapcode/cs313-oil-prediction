import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "ai.oildirection.signalmine",
  appName: "Oil Direction Signal Mine",
  webDir: "dist",
  server: {
    androidScheme: "https",
  },
  android: {
    backgroundColor: "#101418",
  },
};

export default config;
