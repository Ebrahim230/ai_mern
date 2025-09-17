import nodemailer from 'nodemailer';
import dotenv from 'dotenv';

dotenv.config();

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.MAIL_EMAIL,
        pass: process.env.MAIL_SECRET
    }
});

export default ({ to, subject, html }) => {
    return new Promise((resolve, reject) => {
        const options = {
            from: `ThinkestAI <${process.env.MAIL_EMAIL}>`,
            to,
            subject,
            html
        };

        transporter.sendMail(options, (err, info) => {
            if (err) {
                console.error('Email Error:', err);
                reject(err);
            } else {
                console.log('Email sent:', info.response);
                resolve(info);
            }
        });
    });
};
