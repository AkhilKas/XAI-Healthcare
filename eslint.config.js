import js from '@eslint/js';
import react from 'eslint-plugin-react';
import globals from 'globals';

export default [
  {
    ignores: ['build/**', 'node_modules/**', 'dist/**', '.venv/**', '**/__pycache__/**'],
  },
  js.configs.recommended,
  {
    files: ['src/**/*.{js,jsx}', 'tests/frontend/**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.node,
        // Vitest globals (test, expect, describe, vi, etc.)
        ...globals.vitest,
      },
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    plugins: { react },
    rules: {
      ...react.configs.recommended.rules,
      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off',
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
      'no-empty': ['error', { allowEmptyCatch: true }],
    },
    settings: {
      react: { version: 'detect' },
    },
  },
];
