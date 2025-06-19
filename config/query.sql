create extension if not exists "uuid-ossp";

create table users (
  id uuid primary key default uuid_generate_v4(),
  telegram_id text not null,
  role text,
  created_at timestamp with time zone default now()
);
