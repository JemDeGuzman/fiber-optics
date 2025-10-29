// apps/web/web-frontend/__tests__/login.render.test.tsx
import { render, screen } from '@testing-library/react';
import Login from '../src/app/login/page';

describe('Login Page Render Test', () => {
  it('header "LOG IN" found -> Login Page rendered successfully', () => {
    render(<Login />);
    expect(
      screen.getByRole('heading', { name: /LOG IN/i })
    ).toBeInTheDocument();
  });
});
