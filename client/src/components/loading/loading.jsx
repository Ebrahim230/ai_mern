import React from 'react'
import './style.scss'

const Loading = () => {
    return (
        <div data-for='Loading'>
            <div className="inner">
                <h1>ThinkestAI</h1>
                <div data-for="text">Please stand by, while we are checking your browser...</div>
            </div>
        </div>
    )
}

export default Loading
