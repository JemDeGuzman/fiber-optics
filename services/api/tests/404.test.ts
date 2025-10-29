import request from 'supertest';
import app from "../src/app";

describe('404 Not Found', () => {
  it('GET to unknown route -> 404', async () => {
    await request(app).get('/__nope__').expect(404);
  });
});
