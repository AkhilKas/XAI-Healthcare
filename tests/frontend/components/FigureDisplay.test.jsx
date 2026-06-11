import { render, screen } from '@testing-library/react';
import { FigureDisplay } from '../../../src/components/FigureDisplay';

test('returns null when no image data', () => {
  const { container } = render(<FigureDisplay imgBase64={null} />);
  expect(container.firstChild).toBeNull();
});

test('renders img with base64 data URL', () => {
  render(<FigureDisplay imgBase64="ABC123" />);
  const img = screen.getByRole('img', { name: /Generated Figure/i });
  expect(img).toHaveAttribute('src', 'data:image/png;base64,ABC123');
});
