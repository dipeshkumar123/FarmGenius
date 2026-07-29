-- Migration to create the schemes table
DROP TABLE IF EXISTS public.schemes;

CREATE TABLE public.schemes (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    scheme_name TEXT NOT NULL,
    description TEXT NOT NULL,
    eligibility TEXT NOT NULL,
    link TEXT,
    category TEXT NOT NULL,
    amount TEXT,
    states TEXT[] NOT NULL DEFAULT '{"All"}',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security (RLS)
ALTER TABLE public.schemes ENABLE ROW LEVEL SECURITY;

-- Allow all operations for this demo
CREATE POLICY "Allow all operations on schemes"
    ON public.schemes
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Create an index on states for faster filtering
CREATE INDEX IF NOT EXISTS schemes_states_idx ON public.schemes USING GIN (states);
