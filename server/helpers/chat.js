import { db } from "../db/connection.js";
import collections from "../db/collections.js";

export default {
    newResponse: async (prompt, responseText, userId) => {
        try {
            const chatCollection = db.collection(collections.CHAT);
            userId = userId.toString();

            const userDoc = await chatCollection.findOne({ user: userId });

            if (userDoc) {
                await chatCollection.updateOne(
                    { user: userId },
                    { $push: { chats: { prompt, content: responseText } } }
                );
            } else {
                await chatCollection.insertOne({
                    user: userId,
                    chats: [{ prompt, content: responseText }]
                });
            }

            return { success: true };
        } catch (error) {
            throw new Error("Database operation failed");
        }
    },

    updateChat: async (prompt, responseText, userId) => {
        try {
            const chatCollection = db.collection(collections.CHAT);
            userId = userId.toString();

            const result = await chatCollection.updateOne(
                { user: userId },
                { $push: { chats: { prompt, content: responseText } } }
            );

            if (result.matchedCount === 0) {
                throw new Error("User not found");
            }
        } catch (error) {
            throw new Error("Error updating chat");
        }
    },

    getChat: async (userId) => {
        try {
            const result = await db.collection(collections.CHAT).findOne(
                { user: userId.toString() },
                { projection: { _id: 0, chats: 1 } }
            );

            if (!result) throw { status: 404, message: "Chat not found" };
            return result.chats;
        } catch (error) {
            throw error;
        }
    },

    getHistory: async (userId) => {
        try {
            const result = await db.collection(collections.CHAT).findOne(
                { user: userId.toString() },
                { projection: { _id: 0, chats: { $slice: 10 } } }
            );

            if (!result || !result.chats || result.chats.length === 0) {
                return [];
            }

            return result.chats.map((chat, index) => ({
                index,
                prompt: chat.prompt
            }));
        } catch (error) {
            throw new Error("Error fetching chat history");
        }
    },

    deleteAllChat: async (userId) => {
        try {
            await db.collection(collections.CHAT).updateOne(
                { user: userId.toString() },
                { $set: { chats: [] } }
            );
        } catch (error) {
            throw new Error("Error deleting chats");
        }
    }
};