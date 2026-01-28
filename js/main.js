// Typing Animation
const typedTextElement = document.querySelector('.typed-text');
const phrases = [
    'AI Engineer',
    'Systems Architect', 
    'KV-Cache Optimizer',
    'Infrastructure Builder'
];
let phraseIndex = 0;
let charIndex = 0;
let isDeleting = false;

function typeText() {
    const currentPhrase = phrases[phraseIndex];
    
    if (isDeleting) {
        typedTextElement.textContent = currentPhrase.substring(0, charIndex - 1);
        charIndex--;
    } else {
        typedTextElement.textContent = currentPhrase.substring(0, charIndex + 1);
        charIndex++;
    }
    
    let typeSpeed = isDeleting ? 50 : 100;
    
    if (!isDeleting && charIndex === currentPhrase.length) {
        typeSpeed = 2000;
        isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        phraseIndex = (phraseIndex + 1) % phrases.length;
        typeSpeed = 500;
    }
    
    setTimeout(typeText, typeSpeed);
}

document.addEventListener('DOMContentLoaded', typeText);

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Mobile menu toggle
const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const navLinks = document.querySelector('.nav-links');

if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        mobileMenuBtn.classList.toggle('active');
    });
}

// Navbar scroll effect
const navbar = document.querySelector('.navbar');
let lastScroll = 0;

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 50) {
        navbar.style.background = 'rgba(10, 10, 15, 0.95)';
    } else {
        navbar.style.background = 'rgba(10, 10, 15, 0.8)';
    }
    
    lastScroll = currentScroll;
});

// Scroll animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, observerOptions);

document.querySelectorAll('section').forEach(section => {
    section.classList.add('fade-in');
    observer.observe(section);
});

// Blog System
const BLOGS_INDEX = 'blogs/index.json';
const blogContainer = document.getElementById('blog-container');
const blogEmpty = document.getElementById('blog-empty');

async function loadBlogs() {
    try {
        const response = await fetch(BLOGS_INDEX);
        if (!response.ok) throw new Error('Blog index not found');
        
        const blogs = await response.json();
        
        if (blogs.length === 0) {
            blogEmpty.style.display = 'block';
            return;
        }
        
        blogEmpty.style.display = 'none';
        blogContainer.innerHTML = '';
        
        blogs.sort((a, b) => new Date(b.date) - new Date(a.date));
        
        blogs.forEach(blog => {
            const card = createBlogCard(blog);
            blogContainer.appendChild(card);
        });
        
    } catch (error) {
        console.log('No blogs found or error loading:', error);
        blogEmpty.style.display = 'block';
    }
}

function createBlogCard(blog) {
    const card = document.createElement('a');
    card.className = 'blog-card';
    card.href = `blog.html?slug=${blog.slug}`;
    
    const date = new Date(blog.date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    
    card.innerHTML = `
        <div class="blog-date">${date}</div>
        <h3>${blog.title}</h3>
        <p class="blog-excerpt">${blog.excerpt}</p>
        ${blog.tags ? `
            <div class="blog-tags">
                ${blog.tags.map(tag => `<span class="blog-tag">${tag}</span>`).join('')}
            </div>
        ` : ''}
    `;
    
    return card;
}

// Load blogs on page load
if (blogContainer) {
    loadBlogs();
}

// Blog Post Page
async function loadBlogPost() {
    const params = new URLSearchParams(window.location.search);
    const slug = params.get('slug');
    
    if (!slug) {
        window.location.href = 'index.html#blog';
        return;
    }
    
    try {
        const response = await fetch(`blogs/${slug}.md`);
        if (!response.ok) throw new Error('Blog post not found');
        
        const markdown = await response.text();
        const { meta, content } = parseMarkdown(markdown);
        
        document.title = `${meta.title} | Rohith Behera`;
        
        const postContainer = document.getElementById('blog-post-container');
        if (postContainer) {
            const date = new Date(meta.date).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            });
            
            postContainer.innerHTML = `
                <article class="blog-post">
                    <header class="blog-post-header">
                        <h1 class="blog-post-title">${meta.title}</h1>
                        <div class="blog-post-meta">
                            <span>${date}</span>
                            ${meta.readTime ? `<span>${meta.readTime} min read</span>` : ''}
                        </div>
                    </header>
                    <div class="blog-post-content">
                        ${marked.parse(content)}
                    </div>
                </article>
            `;
        }
        
    } catch (error) {
        console.error('Error loading blog post:', error);
        window.location.href = 'index.html#blog';
    }
}

function parseMarkdown(markdown) {
    const lines = markdown.split('\n');
    let inFrontMatter = false;
    let frontMatterLines = [];
    let contentLines = [];
    let frontMatterEnded = false;
    
    for (const line of lines) {
        if (line.trim() === '---' && !frontMatterEnded) {
            if (inFrontMatter) {
                frontMatterEnded = true;
                inFrontMatter = false;
            } else {
                inFrontMatter = true;
            }
            continue;
        }
        
        if (inFrontMatter) {
            frontMatterLines.push(line);
        } else if (frontMatterEnded) {
            contentLines.push(line);
        }
    }
    
    const meta = {};
    frontMatterLines.forEach(line => {
        const colonIndex = line.indexOf(':');
        if (colonIndex > 0) {
            const key = line.substring(0, colonIndex).trim();
            let value = line.substring(colonIndex + 1).trim();
            if (value.startsWith('"') && value.endsWith('"')) {
                value = value.slice(1, -1);
            }
            meta[key] = value;
        }
    });
    
    return {
        meta,
        content: contentLines.join('\n')
    };
}

// Check if on blog post page
if (window.location.pathname.includes('blog.html')) {
    loadBlogPost();
}

// Counter animation for stats
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.textContent = value.toLocaleString() + (element.dataset.suffix || '');
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Animate stats on scroll
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const statValues = entry.target.querySelectorAll('.stat-value');
            statValues.forEach(stat => {
                const text = stat.textContent;
                const numMatch = text.match(/[\d,]+/);
                if (numMatch && !stat.dataset.animated) {
                    const num = parseInt(numMatch[0].replace(/,/g, ''));
                    stat.dataset.animated = 'true';
                    stat.textContent = '0';
                    setTimeout(() => {
                        animateValue(stat, 0, num, 2000);
                        setTimeout(() => {
                            stat.textContent = text;
                        }, 2100);
                    }, 200);
                }
            });
        }
    });
}, { threshold: 0.5 });

const heroStats = document.querySelector('.hero-stats');
if (heroStats) {
    statsObserver.observe(heroStats);
}

