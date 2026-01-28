// Get base path for GitHub Pages
function getBasePath() {
    const path = window.location.pathname;
    if (path.includes('/rohith-portfolio')) {
        return '/rohith-portfolio/';
    }
    return './';
}

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

// Blog System - Index Page
const blogContainer = document.getElementById('blog-container');
const blogEmpty = document.getElementById('blog-empty');

async function loadBlogs() {
    if (!blogContainer) return;
    
    try {
        const basePath = getBasePath();
        const response = await fetch(basePath + 'blogs/index.json');
        if (!response.ok) throw new Error('Blog index not found');
        
        const blogs = await response.json();
        
        if (blogs.length === 0) {
            if (blogEmpty) blogEmpty.style.display = 'block';
            return;
        }
        
        if (blogEmpty) blogEmpty.style.display = 'none';
        blogContainer.innerHTML = '';
        
        blogs.sort((a, b) => new Date(b.date) - new Date(a.date));
        
        blogs.forEach(blog => {
            const card = createBlogCard(blog);
            blogContainer.appendChild(card);
        });
        
    } catch (error) {
        console.log('No blogs found or error loading:', error);
        if (blogEmpty) blogEmpty.style.display = 'block';
    }
}

function createBlogCard(blog) {
    const card = document.createElement('a');
    card.className = 'blog-card';
    const basePath = getBasePath();
    card.href = basePath + 'blog.html?slug=' + blog.slug;
    
    const date = new Date(blog.date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    
    let tagsHtml = '';
    if (blog.tags && blog.tags.length > 0) {
        tagsHtml = '<div class="blog-tags">' + 
            blog.tags.map(tag => '<span class="blog-tag">' + tag + '</span>').join('') + 
            '</div>';
    }
    
    card.innerHTML = 
        '<div class="blog-date">' + date + '</div>' +
        '<h3>' + blog.title + '</h3>' +
        '<p class="blog-excerpt">' + blog.excerpt + '</p>' +
        tagsHtml;
    
    return card;
}

// Blog Post Page
async function loadBlogPost() {
    const postContainer = document.getElementById('blog-post-container');
    if (!postContainer) return;
    
    const params = new URLSearchParams(window.location.search);
    const slug = params.get('slug');
    const basePath = getBasePath();
    
    if (!slug) {
        postContainer.innerHTML = '<div style="padding: 8rem 2rem; text-align: center;"><p>No blog post specified.</p><a href="' + basePath + 'index.html#blog" style="color: #00ffc8;">← Back to Blog</a></div>';
        return;
    }
    
    try {
        const response = await fetch(basePath + 'blogs/' + slug + '.md');
        if (!response.ok) throw new Error('Blog post not found');
        
        const markdown = await response.text();
        const parsed = parseMarkdown(markdown);
        
        document.title = parsed.meta.title + ' | Rohith Behera';
        
        const date = new Date(parsed.meta.date).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        
        let readTimeHtml = '';
        if (parsed.meta.readTime) {
            readTimeHtml = '<span>' + parsed.meta.readTime + ' min read</span>';
        }
        
        postContainer.innerHTML = 
            '<article class="blog-post">' +
                '<header class="blog-post-header">' +
                    '<h1 class="blog-post-title">' + parsed.meta.title + '</h1>' +
                    '<div class="blog-post-meta">' +
                        '<span>' + date + '</span>' +
                        readTimeHtml +
                    '</div>' +
                '</header>' +
                '<div class="blog-post-content">' +
                    marked.parse(parsed.content) +
                '</div>' +
                '<div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #2a2a40;">' +
                    '<a href="' + basePath + 'index.html#blog" style="color: #00ffc8; text-decoration: none;">← Back to Blog</a>' +
                '</div>' +
            '</article>';
        
    } catch (error) {
        console.error('Error loading blog post:', error);
        postContainer.innerHTML = '<div style="padding: 8rem 2rem; text-align: center;"><p>Blog post not found.</p><a href="' + basePath + 'index.html#blog" style="color: #00ffc8;">← Back to Blog</a></div>';
    }
}

function parseMarkdown(markdown) {
    const lines = markdown.split('\n');
    let inFrontMatter = false;
    let frontMatterLines = [];
    let contentLines = [];
    let frontMatterEnded = false;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
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
    frontMatterLines.forEach(function(line) {
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
        meta: meta,
        content: contentLines.join('\n')
    };
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Load blogs on index page
    if (blogContainer) {
        loadBlogs();
    }
    
    // Load blog post on blog page
    if (document.getElementById('blog-post-container')) {
        loadBlogPost();
    }
});

// Counter animation for stats
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = function(timestamp) {
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
const statsObserver = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
        if (entry.isIntersecting) {
            const statValues = entry.target.querySelectorAll('.stat-value');
            statValues.forEach(function(stat) {
                const text = stat.textContent;
                const numMatch = text.match(/[\d,]+/);
                if (numMatch && !stat.dataset.animated) {
                    const num = parseInt(numMatch[0].replace(/,/g, ''));
                    stat.dataset.animated = 'true';
                    stat.textContent = '0';
                    setTimeout(function() {
                        animateValue(stat, 0, num, 2000);
                        setTimeout(function() {
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
