import type { Config } from 'jest';

const config: Config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>'],
  testMatch: ['**/?(*.)+(test|spec).[tj]s?(x)'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'json'],
  setupFiles: ['dotenv/config'],
  globals: { 'ts-jest': { isolatedModules: true } },
};
export default config;
