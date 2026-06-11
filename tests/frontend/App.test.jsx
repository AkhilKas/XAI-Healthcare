import { render, screen } from '@testing-library/react';
import App from '../../src/App';

test('renders the dashboard title in the sidebar', () => {
  render(<App />);
  expect(screen.getByText(/XAI Healthcare/i)).toBeInTheDocument();
  expect(screen.getByText(/Motion Assessment Platform/i)).toBeInTheDocument();
});

test('renders the three view tabs', () => {
  render(<App />);
  expect(screen.getByRole('button', { name: /Glance/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Scan/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Investigate/i })).toBeInTheDocument();
});
