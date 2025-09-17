import { db } from "../db/connection.js";
import collections from "../db/collections.js";
import bcrypt from 'bcrypt';
import { ObjectId } from "mongodb";

export default {
    signup: async ({ email, pass, manual }) => {
        let done = null;
        let userId = new ObjectId().toHexString();

        try {
            let check = await db.collection(collections.USER).findOne({ email });

            if (check) {
                throw { exists: true, text: 'Email already used' };
            }

            pass = await bcrypt.hash(pass, 10);

            await db.collection(collections.TEMP).createIndex({ email: 1 }, { unique: true });
            await db.collection(collections.TEMP).createIndex({ expireAt: 1 }, { expireAfterSeconds: 3600 });

            done = await db.collection(collections.TEMP).insertOne({
                _id: new ObjectId(userId),
                userId: `${userId}_register`,
                email: `${email}_register`,
                register: true,
                pass,
                manual,
                pending: true,
                expireAt: new Date(),
            });
        } catch (err) {
            if (err?.code === 11000) {
                done = await db.collection(collections.TEMP).findOneAndUpdate(
                    { email: `${email}_register`, register: true },
                    { $set: { pass, manual } },
                    { returnDocument: 'after' }
                );
            } else {
                throw err;
            }
        }

        if (done?.value) {
            return { _id: done.value._id.toString(), manual };
        } else if (done?.insertedId) {
            return { _id: done.insertedId.toString(), manual };
        } else {
            throw { exists: true, text: 'Email already used' };
        }
    },

    login: async ({ email, pass, manual }) => {
        const user = await db.collection(collections.USER).findOne({ email });

        if (!user) throw { status: 422, text: 'Invalid login credentials' };

        if (!manual) {
            delete user.pass;
            return user;
        }

        const check = await bcrypt.compare(pass, user.pass);
        if (!check) throw { status: 422, text: 'Invalid login credentials' };

        delete user.pass;
        return user;
    },

    checkUserFound: async (decoded) => {
        return await db.collection(collections.USER).findOne({ _id: new ObjectId(decoded._id) });
    },

    deleteUser: async (userId) => {
        return await db.collection(collections.USER).deleteOne({ _id: new ObjectId(userId) });
    },

    checkPending: async (_id) => {
        if (_id.length !== 24) throw { status: 400, text: 'Invalid ID' };

        return await db.collection(collections.TEMP).findOne({
            _id: new ObjectId(_id),
            pending: true,
        });
    },

    finishSignup: async ({ _id, fName, lName }) => {
        if (_id.length !== 24) throw { status: 400, text: 'Invalid ID' };

        const user = await db.collection(collections.TEMP).findOne({
            _id: new ObjectId(_id),
            pending: true,
        });

        if (!user) throw { status: 404, text: 'User not found or not pending' };

        const newUser = {
            _id: user._id,
            email: user.email.replace('_register', ''),
            UserID: `${fName}${lName}`,
            pass: user.pass,
            manual: user.manual,
            status: 'active',
            createdAt: new Date(),
        };

        await db.collection(collections.USER).insertOne(newUser);
        await db.collection(collections.TEMP).deleteOne({ _id: user._id });

        return newUser;
    },

    generateResetToken: async (email) => {
        const user = await db.collection(collections.USER).findOne({ email });
        if (!user) return null;
    
        const token = crypto.randomBytes(32).toString('hex');
        const expiry = Date.now() + 3600000; // 1 hour
    
        await db.collection(collections.USER).updateOne(
          { _id: user._id },
          { $set: { resetToken: token, resetTokenExpiry: expiry } }
        );
    
        return { _id: user._id, email: user.email, resetToken: token };
      },
    
      resetPasswordWithToken: async ({ token, newPass }) => {
        const now = Date.now();
        const user = await db.collection(collections.USER).findOne({
          resetToken: token,
          resetTokenExpiry: { $gt: now }
        });
    
        if (!user) throw { status: 400, text: "Invalid or expired token" };
    
        const hashedPass = await bcrypt.hash(newPass, 10);
    
        await db.collection(collections.USER).updateOne(
          { _id: user._id },
          { $set: { pass: hashedPass }, $unset: { resetToken: "", resetTokenExpiry: "" } }
        );
    
        return { status: 200, text: "Password reset successful" };
      },
};
