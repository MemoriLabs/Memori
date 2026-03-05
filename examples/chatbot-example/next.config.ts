import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack: (config, { isServer }) => {
    config.resolve = config.resolve ?? {};
    config.resolve.alias = {
      ...config.resolve.alias,
      "@memorilabs/memori": require.resolve("../../memori-ts"),
      "@memorilabs/axon": require.resolve("../../memori-ts/node_modules/@memorilabs/axon"),
    };
    return config;
  },
};

export default nextConfig;
