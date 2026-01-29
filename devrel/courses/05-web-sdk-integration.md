# Course: Web SDK Integration

**Duration**: 12 hours  
**Level**: Intermediate  
**Official Documentation**: https://memorilabs.ai/docs/libraries/web-sdk/overview

---

## Course Overview

The Memori Web SDK brings enterprise AI directly into your web applications. Embed powerful search and chat components into intranet portals, internal tools, and employee-facing applications with minimal code.

### When to Use Web SDK

✅ Building internal web applications  
✅ Embedding search in intranet portals  
✅ Adding AI chat to employee tools  
✅ Creating custom knowledge interfaces  

---

## Module 1: SDK Overview (2 hours)

### Lesson 1.1: Available Components

**Memori Web SDK provides 5 core components:**

1. **Memori Chat** - Full conversational AI interface
2. **Autocomplete + Search Results** - Combined search experience
3. **Modal Search** - Pop-up search interface
4. **Sidebar Search** - Slide-out search panel
5. **Recommendations** - Contextual content suggestions

### Lesson 1.2: Features

**Key Capabilities:**

- **Zero backend required** - Client-side integration
- **Authenticated search** - Respects user permissions
- **Customizable UI** - Themeable components
- **Responsive design** - Works on all devices
- **SSO integration** - Seamless authentication

### Lesson 1.3: Prerequisites

**Requirements:**

- Modern web framework (React, Vue, Angular, or vanilla JS)
- Memori workspace with Web SDK enabled
- User authentication system (SSO recommended)

### Lesson 1.4: Installation

**Install via npm:**

```bash
npm install @memoriwork/web-sdk
```

**Or via CDN:**

```html
<script src="https://cdn.memori.com/web-sdk/v1/memori-web-sdk.js"></script>
```

### 🎯 Lab 1: Setup Environment

**Objective**: Install Web SDK and configure authentication

**Tasks:**
1. Create new React/Vue/HTML project
2. Install Memori Web SDK
3. Configure authentication
4. Verify connection to Memori workspace

---

## Module 2: Authentication & Configuration (2 hours)

### Lesson 2.1: Authentication Methods

**Supported Methods:**

1. **SSO (Recommended)** - SAML, OAuth, OIDC
2. **API Tokens** - For testing/development
3. **JWT** - Custom token-based auth

### Lesson 2.2: SSO Configuration

**Setup SSO with Memori:**

```typescript
import { MemoriSDK } from '@memoriwork/web-sdk';

const memori = new MemoriSDK({
  instance: 'your-company',
  authMode: 'sso',
  ssoConfig: {
    provider: 'okta', // or 'azure', 'google', etc.
    redirectUri: window.location.origin + '/auth/callback'
  }
});

// Initialize
await memori.initialize();
```

### Lesson 2.3: API Token Configuration

**For development/testing:**

```typescript
import { MemoriSDK } from '@memoriwork/web-sdk';

const memori = new MemoriSDK({
  instance: 'your-company',
  authMode: 'token',
  apiToken: process.env.GLEAN_API_TOKEN,
  actAs: 'user@company.com'
});

await memori.initialize();
```

### Lesson 2.4: Third-Party Cookies

**Important**: Web SDK requires third-party cookies for authentication.

**Handle cookie restrictions:**

```typescript
// Detect third-party cookie support
const memori = new MemoriSDK({
  instance: 'your-company',
  authMode: 'sso',
  onCookieError: () => {
    // Show message to enable third-party cookies
    showCookieWarning();
  }
});
```

**Cookie Warning Component:**

```tsx
function CookieWarning() {
  return (
    <div className="cookie-warning">
      <p>Memori requires third-party cookies to be enabled.</p>
      <p>Please enable them in your browser settings:</p>
      <ul>
        <li>Chrome: Settings → Privacy → Third-party cookies</li>
        <li>Firefox: Settings → Privacy → Custom → Cookies</li>
        <li>Safari: Preferences → Privacy → Uncheck "Block all cookies"</li>
      </ul>
    </div>
  );
}
```

### 🎯 Lab 2: Configure Authentication

**Objective**: Set up SSO authentication

**Tasks:**
1. Configure SSO provider (Okta/Azure/Google)
2. Implement auth callback handler
3. Handle authentication errors
4. Test login flow

---

## Module 3: Memori Chat Component (2 hours)

### Lesson 3.1: Basic Chat Integration

**Add Memori Chat to your app:**

```tsx
import { MemoriChat } from '@memoriwork/web-sdk';

function App() {
  return (
    <div className="app">
      <h1>Company Portal</h1>
      
      <MemoriChat
        instance="your-company"
        height="600px"
        width="100%"
      />
    </div>
  );
}
```

### Lesson 3.2: Chat Customization

**Customize appearance:**

```tsx
<MemoriChat
  instance="your-company"
  theme={{
    primaryColor: '#1a73e8',
    backgroundColor: '#ffffff',
    fontFamily: 'Inter, sans-serif'
  }}
  placeholder="Ask me anything about our company..."
  showSources={true}
  enableVoice={true}
/>
```

### Lesson 3.3: Chat Events

**Handle chat events:**

```tsx
<MemoriChat
  instance="your-company"
  onMessageSent={(message) => {
    console.log('User sent:', message);
    trackAnalytics('chat_message', { query: message });
  }}
  onResponseReceived={(response) => {
    console.log('Memori responded:', response);
  }}
  onError={(error) => {
    console.error('Chat error:', error);
    showErrorNotification(error.message);
  }}
/>
```

### Lesson 3.4: Programmatic Control

**Control chat programmatically:**

```tsx
import { useRef } from 'react';
import { MemoriChat } from '@memoriwork/web-sdk';

function App() {
  const chatRef = useRef();
  
  const askQuestion = (question) => {
    chatRef.current?.sendMessage(question);
  };
  
  const clearChat = () => {
    chatRef.current?.clearHistory();
  };
  
  return (
    <div>
      <button onClick={() => askQuestion('What is our PTO policy?')}>
        Quick Question
      </button>
      <button onClick={clearChat}>
        Clear History
      </button>
      
      <MemoriChat
        ref={chatRef}
        instance="your-company"
      />
    </div>
  );
}
```

### 🎯 Lab 3: Build Chat Interface

**Objective**: Implement customized chat component

**Requirements:**
1. Add Memori Chat to webpage
2. Customize theme to match company branding
3. Add event tracking
4. Implement quick actions (pre-filled questions)
5. Handle errors gracefully

---

## Module 4: Search Components (3 hours)

### Lesson 4.1: Autocomplete + Search Results

**Full search page:**

```tsx
import { MemoriSearch } from '@memoriwork/web-sdk';

function SearchPage() {
  return (
    <div className="search-page">
      <header>
        <h1>Company Search</h1>
      </header>
      
      <MemoriSearch
        instance="your-company"
        variant="full"
        showFilters={true}
        resultsPerPage={20}
      />
    </div>
  );
}
```

**Customization:**

```tsx
<MemoriSearch
  instance="your-company"
  variant="full"
  theme={{
    primaryColor: '#ff6b35',
    resultCardStyle: 'compact'
  }}
  datasources={['confluence', 'google-drive', 'slack']}
  placeholder="Search documents, people, and conversations..."
  showFilters={true}
  enableAutocomplete={true}
/>
```

### Lesson 4.2: Modal Search

**Pop-up search:**

```tsx
import { useState } from 'react';
import { MemoriSearchModal } from '@memoriwork/web-sdk';

function App() {
  const [searchOpen, setSearchOpen] = useState(false);
  
  // Keyboard shortcut (Cmd+K / Ctrl+K)
  useEffect(() => {
    const handleKeyboard = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen(true);
      }
    };
    
    document.addEventListener('keydown', handleKeyboard);
    return () => document.removeEventListener('keydown', handleKeyboard);
  }, []);
  
  return (
    <div>
      <button onClick={() => setSearchOpen(true)}>
        🔍 Search (⌘K)
      </button>
      
      <MemoriSearchModal
        instance="your-company"
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
        onResultClick={(result) => {
          window.open(result.url, '_blank');
          setSearchOpen(false);
        }}
      />
    </div>
  );
}
```

### Lesson 4.3: Sidebar Search

**Slide-out search panel:**

```tsx
import { MemoriSearchSidebar } from '@memoriwork/web-sdk';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  return (
    <div className="app">
      <header>
        <button onClick={() => setSidebarOpen(!sidebarOpen)}>
          Toggle Search
        </button>
      </header>
      
      <main>
        {/* Your main content */}
      </main>
      
      <MemoriSearchSidebar
        instance="your-company"
        open={sidebarOpen}
        side="right"
        width="400px"
        onClose={() => setSidebarOpen(false)}
      />
    </div>
  );
}
```

### Lesson 4.4: Search Results Handling

**Custom result rendering:**

```tsx
<MemoriSearch
  instance="your-company"
  onResultClick={(result) => {
    // Track analytics
    trackAnalytics('search_result_click', {
      title: result.title,
      datasource: result.datasource,
      url: result.url
    });
    
    // Custom navigation
    navigateToResult(result);
  }}
  renderResult={(result) => (
    <div className="custom-result">
      <img src={result.thumbnail} alt="" />
      <div>
        <h3>{result.title}</h3>
        <p>{result.snippet}</p>
        <span className="datasource">{result.datasource}</span>
      </div>
    </div>
  )}
/>
```

### 🎯 Lab 4: Build Search Experience

**Objective**: Implement all search components

**Tasks:**
1. Create full search page with autocomplete
2. Add modal search with keyboard shortcut
3. Implement sidebar search
4. Customize result cards
5. Add search analytics

---

## Module 5: Recommendations Component (2 hours)

### Lesson 5.1: Basic Recommendations

**Show contextual recommendations:**

```tsx
import { MemoriRecommendations } from '@memoriwork/web-sdk';

function DocumentPage({ documentId }) {
  return (
    <div className="document-page">
      <article>
        {/* Document content */}
      </article>
      
      <aside>
        <h3>Related Content</h3>
        <MemoriRecommendations
          instance="your-company"
          context={{
            documentId,
            type: 'related-documents'
          }}
          maxResults={5}
        />
      </aside>
    </div>
  );
}
```

### Lesson 5.2: Personalized Recommendations

**Based on user activity:**

```tsx
<MemoriRecommendations
  instance="your-company"
  context={{
    type: 'personalized',
    userRole: 'engineer',
    recentActivity: ['api-docs', 'architecture']
  }}
  title="Recommended for You"
  maxResults={10}
/>
```

### Lesson 5.3: Topic-Based Recommendations

**Show recommendations by topic:**

```tsx
<MemoriRecommendations
  instance="your-company"}
  context={{
    type: 'topic',
    topic: 'product-launch',
    department: 'marketing'
  }}
  layout="grid"
  cardStyle="detailed"
/>
```

### 🎯 Lab 5: Add Recommendations

**Objective**: Implement recommendation system

**Tasks:**
1. Add related content recommendations
2. Show personalized suggestions on homepage
3. Implement topic-based recommendations
4. Customize recommendation cards

---

## Module 6: Advanced Integration (1 hour)

### Lesson 6.1: Multi-Component Layout

**Combine components:**

```tsx
function CompanyPortal() {
  const [chatOpen, setChatOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  
  return (
    <div className="portal">
      <header>
        <Logo />
        <SearchBar onFocus={() => setSearchOpen(true)} />
        <button onClick={() => setChatOpen(true)}>
          Ask AI
        </button>
      </header>
      
      <main>
        <MemoriRecommendations
          instance="your-company"
          context={{ type: 'personalized' }}
        />
      </main>
      
      <MemoriSearchModal
        instance="your-company"
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
      />
      
      <MemoriChat
        instance="your-company"
        open={chatOpen}
        mode="floating"
        position="bottom-right"
        onClose={() => setChatOpen(false)}
      />
    </div>
  );
}
```

### Lesson 6.2: Analytics Integration

**Track usage:**

```typescript
import { MemoriSDK } from '@memoriwork/web-sdk';

const memori = new MemoriSDK({
  instance: 'your-company',
  onEvent: (event) => {
    // Send to analytics platform
    analytics.track(event.type, {
      component: event.component,
      data: event.data,
      userId: event.userId,
      timestamp: event.timestamp
    });
  }
});
```

### Lesson 6.3: Error Handling

**Global error handling:**

```typescript
const memori = new MemoriSDK({
  instance: 'your-company',
  onError: (error) => {
    console.error('Memori SDK Error:', error);
    
    // Log to error tracking
    Sentry.captureException(error);
    
    // Show user-friendly message
    if (error.code === 'AUTH_FAILED') {
      showNotification('Please sign in again');
    } else if (error.code === 'NETWORK_ERROR') {
      showNotification('Connection issue. Please try again.');
    }
  }
});
```

### 🎯 Final Project: Complete Portal

**Objective**: Build a comprehensive company portal

**Requirements:**
1. Homepage with personalized recommendations
2. Full search page with filters
3. Modal search (⌘K shortcut)
4. Floating chat assistant
5. Analytics tracking
6. Error handling
7. Custom branding/theme
8. Mobile responsive

**Deliverables:**
- Working web application
- Source code (GitHub)
- Documentation
- Demo video (8 minutes)

---

## Assessment

### Quiz (15 questions)
1. What are the 5 Web SDK components?
2. How do you handle third-party cookie restrictions?
3. What authentication methods are supported?
4. How do you customize component themes?
5. How do you track component events?

### Practical Assessment

**Build an Internal Knowledge Portal**

**Requirements:**
- Use at least 3 Web SDK components
- Implement SSO authentication
- Add custom branding
- Track user interactions
- Handle errors gracefully
- Mobile responsive design
- Accessibility compliant (WCAG 2.1)

**Submission:**
- GitHub repository
- Live demo URL
- Setup documentation
- Demo video (8 minutes)
- Analytics report

---

## Additional Resources

- **Web SDK Docs**: https://memorilabs.ai/docs/libraries/web-sdk/overview
- **API Reference**: https://memorilabs.ai/docs/libraries/web-sdk/api
- **Component Playground**: https://sdk-playground.memori.com
- **GitHub Examples**: https://github.com/memoriwork/web-sdk-examples
- **Support**: support@memori.com

---

## Next Steps

1. Take the **Memori Integration Specialist** certification
2. Explore **Direct API** for backend integrations
3. Learn **Agent Toolkit** for AI agents
4. Build production web applications

**Certificate**: Upon completion, you'll receive a **Memori Web SDK Integration Certificate**
