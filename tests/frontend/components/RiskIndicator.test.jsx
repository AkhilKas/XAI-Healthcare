import { render, screen } from '@testing-library/react';
import { RiskIndicator } from '../../../src/components/RiskIndicator';

test('renders label and percentage value', () => {
  render(<RiskIndicator label="Range of Motion" value={75} color="#22c55e" />);
  expect(screen.getByText('Range of Motion')).toBeInTheDocument();
  expect(screen.getByText('75%')).toBeInTheDocument();
});

test('applies color prop to the progress arc', () => {
  const { container } = render(<RiskIndicator label="MQ" value={50} color="#ef4444" />);
  const circles = container.querySelectorAll('circle');
  // second circle is the colored arc
  expect(circles[1]).toHaveAttribute('stroke', '#ef4444');
});

test('encodes value into strokeDasharray', () => {
  const { container } = render(<RiskIndicator label="MQ" value={50} color="#000" />);
  const circles = container.querySelectorAll('circle');
  const [dash, gap] = circles[1].getAttribute('stroke-dasharray').split(' ');
  expect(parseFloat(dash)).toBeCloseTo(100.5, 1);
  expect(gap).toBe('201');
});
