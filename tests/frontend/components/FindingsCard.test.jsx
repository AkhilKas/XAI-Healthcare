import { render, screen } from '@testing-library/react';
import { FindingsCard } from '../../../src/components/FindingsCard';

test.each([
  ['error', 'bg-red-50', 'border-red-200'],
  ['warning', 'bg-amber-50', 'border-amber-200'],
  ['success', 'bg-green-50', 'border-green-200'],
])('applies %s styling', (type, bg, border) => {
  const { container } = render(
    <FindingsCard type={type} title={`${type} title`} description="desc" />
  );
  const root = container.firstChild;
  expect(root).toHaveClass(bg);
  expect(root).toHaveClass(border);
});

test('renders title and description', () => {
  render(<FindingsCard type="error" title="Scapular rhythm" description="dyskinesis" />);
  expect(screen.getByText('Scapular rhythm')).toBeInTheDocument();
  expect(screen.getByText('dyskinesis')).toBeInTheDocument();
});
