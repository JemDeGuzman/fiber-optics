import request from 'supertest';
import app from "../src/app";

describe('Health/Ping check', () => {
  it('GET /api/ping -> 200 with status ok', async () => {
    const res = await request(app).get('/api/ping');
    expect(res.status).toBe(200);
    expect(res.body).toEqual(expect.objectContaining({ ok: true, ts: expect.any(Number) }));
  });
});