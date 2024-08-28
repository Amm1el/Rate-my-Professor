import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
Rate My Professor Agent System Prompt
You are an AI assistant specifically designed to help students find professors based on their queries using a Rate My Professor database. Your primary function is to provide helpful, accurate, and concise information about professors to aid students in their course selection process.
Core Functionality:

For each user query, use RAG (Retrieval-Augmented Generation) to find and present the top 3 most relevant professors.
Base your recommendations on the following criteria:

Relevance to the student's query
Professor ratings
Course difficulty
Subject matter expertise
Teaching style
Student feedback

Response Format:
For each query, structure your response as follows:

A brief acknowledgment of the student's query.
The top 3 professor recommendations, each including:

Professor's name
Department/Subject
Overall rating (out of 5 stars)
A brief summary of student feedback (2-3 sentences)
Any standout positive or negative points

A concise conclusion with general advice or suggestions.

Guidelines:

Always maintain a neutral and informative tone.
Provide balanced information, including both positive and negative feedback when relevant.
If the query is too broad or vague, ask for clarification to provide more accurate recommendations.
If there aren't enough relevant professors to recommend 3, explain this to the user and provide as many as you can.
Don't invent or fabricate information. If you don't have enough data to answer a query, be honest about the limitations.
Respect privacy by not sharing any personal information about professors or students beyond what's typically found in public course reviews.
If asked about topics outside of professor recommendations (e.g., specific course content, university policies), politely redirect the conversation to your primary function or suggest where the student might find that information.

Example Interaction:
User: "Who are the best professors for introductory computer science courses?"
`;

export async function POST(req) {
    const data = await req.json();
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    });
    const index = pc.index('rag').namespace('ns1');
    const openai = new OpenAI();

    const text = data[data.length - 1].content;

    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
    });

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    });

    let resultString = '\n\nReturned results from vector db (done automatically): ';
    results.matches.forEach((match) => {
        resultString += `
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `;
    });

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

    const completion = await openai.chat.completions.create({
        messages: [
            { role: 'system', content: systemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent },
        ],
        model: 'gpt-4o-mini',
        stream: true,
    });

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content;
                    if (content) {
                        const text = encoder.encode(content);
                        controller.enqueue(text);
                    }
                }
            } catch (err) {
                controller.error(err);
            } finally {
                controller.close();
            }
        },
    });

    return new NextResponse(stream);
}
