import { render, screen } from '@testing-library/react';
import { BodyDiagram } from '../../../src/components/BodyDiagram';

const RED = '#ef4444';
const YELLOW = '#fbbf24';
const GREEN = '#22c55e';

test('colors head/arms by severity thresholds', () => {
  const { container } = render(
    <BodyDiagram values={{ head: 80, left: 50, right: 10 }} />
  );
  const head = container.querySelector('ellipse');
  const arms = container.querySelectorAll('line[stroke-linecap="round"]');
  expect(head).toHaveAttribute('fill', RED);
  expect(arms[0]).toHaveAttribute('stroke', YELLOW); // left
  expect(arms[1]).toHaveAttribute('stroke', GREEN);  // right
});

test('renders left/right shoulder labels', () => {
  render(<BodyDiagram values={{ head: 0, left: 0, right: 0 }} />);
  expect(screen.getByText('Left shoulder')).toBeInTheDocument();
  expect(screen.getByText('Right shoulder')).toBeInTheDocument();
});

test('uses default values when none passed', () => {
  const { container } = render(<BodyDiagram />);
  expect(container.querySelector('svg')).toBeInTheDocument();
});
